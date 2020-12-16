import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Domain:
    def __init__(self, shape, e, w, c, tau):

        self.dim = len(shape)
        self.d = len(e)
        self.e = e
        self.w = w
        self.c = c
        self.omega = 1. / tau

        shape.append(self.d)
        self.f = numpy.empty(shape, dtype=numpy.float64)
        self.fs = self.f.reshape(-1, self.d)

    def _state(self, f):
        density = sum(f)
        velocity = (f @ self.e) * self.c / density
        return density, velocity

    def _equilibrium(self, rho, u):
        ux = self.c * self.e @ u
        return rho * self.w * (1 + ux + .5 * (ux * ux - numpy.dot(u, u)))

    def _stream(self):
        for i, ei in enumerate(self.e):
            self.f[..., i] = numpy.roll(self.f[..., i], ei, range(self.dim))

    def _collide(self):
        for f in self.fs:
            rho, u = self._state(f)
            feq = self._equilibrium(rho, u)
            f -= self.omega * (f - feq)

    def initialize(self, initializer):
        shape = self.f.shape
        for i in range(shape[0]):
            x = float(i) / shape[0]
            for j in range(shape[1]):
                y = float(j) / shape[1]
                rho, u = initializer(x, y)
                self.f[i, j, :] = self._equilibrium(rho, u)

    def advance(self, steps):
        for i in range(steps):
            self._stream()
            self._collide()

    def state(self):
        shape = list(self.f.shape)
        shape[-1] = self.dim + 1
        state = numpy.empty(shape, dtype=numpy.float64)
        ss = state.reshape(-1, self.dim+1)
        for i, f in enumerate(self.fs):
            rho, u = self._state(f)
            ss[i, 0] = rho
            ss[i, 1:] = u

        return state

d2q9_e = numpy.array([
    [ 0,  0],
    [ 1,  0],
    [ 0,  1],
    [-1,  0],
    [ 0, -1],
    [ 1,  1],
    [-1,  1],
    [-1, -1],
    [ 1, -1]
])

d2q9_w = numpy.concatenate(([4./9.], 4 * [1./9.], 4 * [1./36]))

d2q9_c = math.sqrt(3.)

def shear_layer(x, y):
    vx = u0 * math.tanh((abs(y - 0.5) - 0.25) / thickness)
    vy = delta * u0 * math.sin(2. * math.pi * (x - .25))
    return 1., numpy.array([vx, vy])

Reynolds  = 10000.
Mach      = 0.1
L         = [100, 100]
thickness = 0.0125
delta     = 0.05

u0 = Mach 
nu = u0 * L[1] * d2q9_c / Reynolds
tau = nu + .5
#
# setup wave numbers for Fourier transform
#
LH = int(L[0] / 2)
scale = 2j * numpy.pi / L[0]

kx = scale * (numpy.arange(0, L[0]) - LH + 1)
ky = scale * numpy.arange(0, LH + 1)

kx = numpy.roll(kx, LH + 1)
#
# setup computational domain
#
domain = Domain(L, d2q9_e, d2q9_w, d2q9_c, tau)
domain.initialize(shear_layer)
for i in range(200):
    print(i)
    domain.advance(10)

    state = domain.state()

    ux = numpy.fft.rfft2(state[:, :, 1])
    uy = numpy.fft.rfft2(state[:, :, 2])

    vorticity = numpy.fft.irfft2(ux * ky - (uy.T * kx).T)

    plt.imsave("w%d.png" % (i), vorticity.T, cmap=cm.gray, origin='lower')