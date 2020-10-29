import numpy as np
import utils
import cv2 as cv


class ParticleSystem:
    def __init__(self, dim_xy, N):
        self.dim_xy = dim_xy + []
        self.N = N
        self.focal = 1
        z_max = 4
        dx, dy = 2*np.array(self.dim_xy)/np.max(self.dim_xy)
        self.span_xyz = np.array([[dx, dy, z_max]])
        self.z_offset = 0.01
        self.bounds_xyz = np.array([[-0.5, 0.5], [-0.5, 0.5], [0, 1]]) * self.span_xyz.T
        self.bounds_xyz[2, :] += self.z_offset
        self.center_xy = (self.bounds_xyz[:2, 1] - self.bounds_xyz[:2, 0]).T / 2

        self.pos_xyz = self.space_mod(np.random.rand(N, 3)*self.span_xyz)
        self.screen = np.zeros(shape=dim_xy[::-1], dtype=np.uint8)
        self.update([0, 0, 0], 0)

    def space_mod(self, pos_xyz):
        lower_bounds = self.bounds_xyz[:, 0]
        pos_xyz = np.mod(pos_xyz - lower_bounds, self.span_xyz) + lower_bounds
        return pos_xyz

    def update(self, velocity_xyz, dt):
        """
        Update stars position and render on screen
        :param velocity_xyz: 
        :param dt: 
        :return: 
        """
        self.pos_xyz -= np.array(velocity_xyz) * dt
        self.pos_xyz = self.space_mod(self.pos_xyz)

        # project point on the plane XY
        pos_plan_xy = self.pos_xyz[:, 0:2] * self.focal/self.pos_xyz[:, 2:]

        # plane XY to screen
        lower_bounds_xy = self.bounds_xyz[0:2, 0]
        plane_to_screen_factor = self.dim_xy/self.span_xyz[0, 0:2]
        pos_screen_xy = (pos_plan_xy - lower_bounds_xy) * plane_to_screen_factor
        pos_screen_xy = pos_screen_xy.astype(np.int64)

        # keep only visible points
        low_x, up_x = self.bounds_xyz[0, :]
        low_y, up_y = self.bounds_xyz[1, :]
        visible = np.array([low_x < x < up_x and low_y < y < up_x for x, y, in pos_plan_xy])
        pos_plan_xy = pos_plan_xy[visible]
        pos_screen_xy = pos_screen_xy[visible]

        # brightness decrease with distance
        pos_xyz = self.pos_xyz[visible]
        distance_to_viewer = np.sum(pos_xyz ** 2, axis=1) ** 0.5
        dist_threshold = self.z_offset+self.span_xyz[0, 2]
        brightness = dist_threshold-distance_to_viewer
        brightness[brightness < 0] = 0
        brightness *= 255/dist_threshold

        # stars on the center of the zoom shouldn't be visible
        brightness *= 1-np.exp(-10*np.sum(pos_plan_xy**2, axis=1))
        brightness = brightness.astype(np.uint8)
        # print(max(brightness))

        # radius decrease with distance
        radius = 1 / distance_to_viewer

        # render on screen
        self.screen = (self.screen.astype(float)*0.8).astype(np.uint8)
        for i in range(len(pos_screen_xy)):
            x, y = pos_screen_xy[i]
            r = int(radius[i]/5+1)
            b = int(brightness[i])
            cv.circle(self.screen, (x, y), r, (b,))


if __name__ == '__main__':
    ps = ParticleSystem([500, 300], 1000)
    fps = 60
    velocity_xyz = [0, 0, 1]
    for k in range(200):
        utils.imshow(ps.screen, ms=10)
        ps.update(velocity_xyz, 0.1/fps)

