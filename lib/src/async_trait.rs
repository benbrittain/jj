#[cfg(feature = "std")]
pub use tokio::io::AsyncRead;
#[cfg(feature = "std")]
pub use tokio::io::AsyncReadExt;

#[cfg(not(feature = "std"))]
mod no_std {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::future::Future;
    use core::iter;
    use core::ops::DerefMut;
    use core::pin::Pin;
    use core::task::ready;
    use core::task::Context;
    use core::task::Poll;

    /// Future for the [`read_to_end`](super::AsyncReadExt::read_to_end) method.
    #[derive(Debug)]
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct ReadToEnd<'a, R: ?Sized> {
        reader: &'a mut R,
        buf: &'a mut Vec<u8>,
        start_len: usize,
    }

    impl<'a, R: AsyncRead + ?Sized + Unpin> ReadToEnd<'a, R> {
        pub(super) fn new(reader: &'a mut R, buf: &'a mut Vec<u8>) -> Self {
            let start_len = buf.len();
            Self {
                reader,
                buf,
                start_len,
            }
        }
    }

    impl<A> Future for ReadToEnd<'_, A>
    where
        A: AsyncRead + ?Sized + Unpin,
    {
        type Output = no_std_io::io::Result<usize>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let this = &mut *self;
            read_to_end_internal(Pin::new(&mut this.reader), cx, this.buf, this.start_len)
        }
    }

    struct Guard<'a> {
        buf: &'a mut Vec<u8>,
        len: usize,
    }

    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            #[allow(unsafe_code)]
            unsafe {
                self.buf.set_len(self.len);
            }
        }
    }

    // This uses an adaptive system to extend the vector when it fills. We want to
    // avoid paying to allocate and zero a huge chunk of memory if the reader only
    // has 4 bytes while still making large reads if the reader does have a ton
    // of data to return. Simply tacking on an extra DEFAULT_BUF_SIZE space every
    // time is 4,500 times (!) slower than this if the reader has a very small
    // amount of data to return.
    //
    // Because we're extending the buffer with uninitialized data for trusted
    // readers, we need to make sure to truncate that if any of this panics.
    pub(super) fn read_to_end_internal<R: AsyncRead + ?Sized>(
        mut rd: Pin<&mut R>,
        cx: &mut Context<'_>,
        buf: &mut Vec<u8>,
        start_len: usize,
    ) -> Poll<no_std_io::io::Result<usize>> {
        let mut g = Guard {
            len: buf.len(),
            buf,
        };
        loop {
            if g.len == g.buf.len() {
                g.buf.reserve(32);
                let spare_capacity = g.buf.capacity() - g.buf.len();
                g.buf.extend(iter::repeat(0).take(spare_capacity));
            }

            let buf = &mut g.buf[g.len..];
            match ready!(rd.as_mut().poll_read(cx, buf)) {
                Ok(0) => return Poll::Ready(Ok(g.len - start_len)),
                Ok(n) => {
                    assert!(n <= buf.len());
                    g.len += n;
                }
                Err(e) => return Poll::Ready(Err(e)),
            }
        }
    }

    impl<R: ?Sized + Unpin> Unpin for ReadToEnd<'_, R> {}

    pub trait AsyncRead {
        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<no_std_io::io::Result<usize>>;
    }

    macro_rules! deref_async_read {
        () => {
            fn poll_read(
                mut self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<no_std_io::io::Result<usize>> {
                Pin::new(&mut **self).poll_read(cx, buf)
            }
        };
    }

    impl<T: ?Sized + AsyncRead + Unpin> AsyncRead for Box<T> {
        deref_async_read!();
    }

    impl<T: ?Sized + AsyncRead + Unpin> AsyncRead for &mut T {
        deref_async_read!();
    }

    impl<P> AsyncRead for Pin<P>
    where
        P: DerefMut + Unpin,
        P::Target: AsyncRead,
    {
        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<no_std_io::io::Result<usize>> {
            self.get_mut().as_mut().poll_read(cx, buf)
        }
    }

    macro_rules! delegate_async_read_to_stdio {
        () => {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<no_std_io::io::Result<usize>> {
                Poll::Ready(no_std_io::io::Read::read(&mut *self, buf))
            }
        };
    }

    impl AsyncRead for &[u8] {
        delegate_async_read_to_stdio!();
    }

    // NOTE: this is more like futures::io:: than tokio::io
    pub trait AsyncReadExt: AsyncRead {
        fn read_to_end<'a>(&'a mut self, buf: &'a mut Vec<u8>) -> ReadToEnd<'a, Self>
        where
            Self: Unpin,
        {
            ReadToEnd::new(self, buf)
        }
    }

    impl<R: AsyncRead + ?Sized> AsyncReadExt for R {}
}

#[cfg(not(feature = "std"))]
pub use no_std::*;
