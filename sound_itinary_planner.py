import json
import time
from utils import pth
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


class MusicMap:
    """
    ABSTRACT CLASS
    """

    # folder containing all sidechains tracks and the locations.json
    SOUNDFOLDER = None
    LOCATIONS_JSON_FILE = None
    BPM = None
    # list of names of sidechan tracks such that SOUNDFOLDER/SIDECHAIN_NAME.wav exists
    SIDECHAIN_NAMES = None
    # beat on which the locations must fall on
    LOCATIONS_BEATS = []

    # make sure the child class is like a constant structure
    def __setattr__(self, key, value):
        print("Setting attribute to object of type", self, "is forbidden")
        raise Exception

class AnotherPlanetMap(MusicMap):
    SOUNDFOLDER = "sounds"
    LOCATIONS_JSON_FILE = SOUNDFOLDER + "/locations.json"
    BPM = 100
    SIDECHAIN_NAMES = [
        "ALL",
        "BACKLEAD",
        "BASS",
        "DRUM",
        "GHOST",
        "LEAD",
        "PAFF",
        "SOUNDFX",
        "TEXTURE",
    ]
    # beat on which the locations must fall on
    LOCATIONS_BEATS = 4*np.array([
        1,  # intro
        19,  # A
        43,  # Bridge AB
        47,  # B
        59,  # Bridge BA
        63,  # A
        87,  # Bridge AB
        91,  # B
        103,  # Bridge BA outro
        113,
    ])


class Itinary:
    def __init__(self, music_map, fps):
        """
        :param music_map: object of type MusicMap
        :param fps: frame per seconds of the output video
        """
        self.music_map = music_map
        self.fps = fps

        self.frame_nb = None
        self.compute_frame_nb()

        self.locations = None
        self.load_locations()

        self.sidechains = None
        self.load_sidechain()

        self.full_itinary = None
        self.sparse_itinary = None
        self.generate_itinaries()
        print("ok")

    def compute_frame_nb(self):
        """
        given an audio file name and frame per second (fps),
        returns the number of frame needed to cover the entire soundifle
        :param sound_filename: name of wav file in SOUNDFOLDER
        :param fps:
        :return:
        """

        filepath = f"{self.music_map.SOUNDFOLDER}/{self.music_map.SIDECHAIN_NAMES[0]}.wav"
        fs, soundwave = scipy.io.wavfile.read(filepath)

        nb_channel, nb_samples = soundwave.shape
        nb_sample_per_frame = int(round(fs / self.fps))
        self.frame_nb = (nb_samples // nb_sample_per_frame)

    def load_sidechain(self):
        """
        Returns the volume and trigger channel of the corresponding sound file.
        The volume channel is normalize in [0, 1]
        The trigger channel is 1 at positions where the volume is a local maximum. It is 0 otherwise
        volume and trigger are 1d arrays. Their length is the same as the number of frames contained
        in the output video.

        :param name: name of the sidechain (a .wav file with this name must exist in the SOUNDFOLDER)
        :param fps: frame per second of the target video
        :return: volume, trigger
        """

        def smooth_filter(size):
            out = np.linspace(-5, 5, size)
            out = np.exp(-out**2)
            out /= np.sum(out)
            return out

        self.sidechains = {}
        for name in self.music_map.SIDECHAIN_NAMES:
            filepath = f"{self.music_map.SOUNDFOLDER}/{name}.wav"
            fs, soundwave = scipy.io.wavfile.read(filepath)

            # extract the command (volume) from the soundwave
            volume = np.max(np.abs(soundwave), axis=1)
            # volume = np.convolve(volume, smooth_filter(fs//15))
            block_size = int(round(fs / self.fps))
            length = len(volume)
            block_nb = (length // block_size)
            volume = volume[:block_nb * block_size]
            volume = np.reshape(volume, newshape=(block_nb, block_size))
            volume = np.max(volume, axis=1)
            # normalize command in [0, 1]
            volume = volume.astype(np.float) / (2 ** 15 - 1)

            # extract the trigger points from the volume channel
            trigger = np.zeros_like(volume)
            vmiddle = volume[0]
            vright = volume[1]
            for k in range(1, len(volume)-1):
                vleft = vmiddle
                vmiddle = vright
                vright = volume[k+1]
                if vmiddle/max(vleft, vright)>2:
                    trigger[k] = 1

            # plt.plot(volume, 'g')
            # plt.stem(trigger)
            # plt.show()

            # store the result
            self.sidechains[name] = {}
            self.sidechains[name]["volume"] = volume
            self.sidechains[name]["trigger"] = trigger


    def load_locations(self):
        with open(self.music_map.LOCATIONS_JSON_FILE, "r") as json_in:
            self.locations = json.load(json_in)

        if len(self.locations) != len(self.music_map.LOCATIONS_BEATS):
            print(f"ERROR Number of locations in {self.music_map.LOCATIONS_JSON_FILE} "
                  f"doesn't match self.music_map.LOCATIONS_BEATS: "
                  f"{len(self.locations)} != {len(self.music_map.LOCATIONS_BEATS)}")
            raise Exception

        # convert rotation matrix to np array
        for k in range(len(self.locations)):
            self.locations[k]["r_mat"] = np.array(self.locations[k]["r_mat"])

    def generate_itinaries(self):
        print("generate_itinaries")
        tic0 = time.time()
        bps = self.music_map.BPM/60
        locations_at_frame = [int(round(self.fps*(beat-1)/bps)) for beat in self.music_map.LOCATIONS_BEATS]

        k = 0
        self.full_itinary = []
        self.sparse_itinary = []
        prev_loc = self.locations[0]
        for i in range(len(self.locations) - 1):
            locA = self.locations[i]
            locB = self.locations[i + 1]
            frameA = locations_at_frame[i]
            frameB = locations_at_frame[i + 1]
            nb_inter_frame = frameB - frameA
            for j in range(nb_inter_frame):
                t = j / nb_inter_frame
                loc = Itinary.interpolate_locations(locA, locB, t)

                sc_frames = {}
                for name in self.sidechains.keys():
                    sc_frames.update({
                        name: {
                            "volume": self.sidechains[name]["volume"][k],
                            "trigger": self.sidechains[name]["trigger"][k],
                        }
                    })
                loc["sidechains"] = sc_frames

                xyz0 = np.append(loc["pos_julia_xy"], loc["zoom"])
                xyz1 = np.append(prev_loc["pos_julia_xy"], prev_loc["zoom"])
                velocity_vector = xyz1 - xyz0
                loc["velocity_vector"] = velocity_vector

                self.full_itinary.append(loc)
                if j == 0:
                    self.sparse_itinary.append(loc)
                prev_loc = loc.copy()
                k += 1
        self.sparse_itinary.append(loc)

    @staticmethod
    def interpolate_locations(locA, locB, t):
        out = {}
        for keyword in ["pos_julia_xy"]:
            x = locA[keyword][0] * (1 - t) + t * locB[keyword][0]
            y = locA[keyword][1] * (1 - t) + t * locB[keyword][1]
            out[keyword] = x, y

        for keyword in ["pos_mandel_xy"]:
            dz = (locB["zoom"] - locA["zoom"])*1
            # print(dz)
            if dz == 0:
                t2 = t
            else:
                t2 = (1-2**(-t*dz))/(1-2**(-dz))

            x = locA[keyword][0] * (1 - t2) + t2 * locB[keyword][0]
            y = locA[keyword][1] * (1 - t2) + t2 * locB[keyword][1]
            out[keyword] = x, y

        for keyword in ["zoom", "fisheye_factor"]:  # , "r_mat",]:
            out[keyword] = locA[keyword] * (1 - t) + t * locB[keyword]
        out["r_mat"] = np.eye(4)
        return out

    @staticmethod
    def get_smooth_trigger_from_full_itinary(full_itinary, sidechain_name, fps, attack_s=0.1, release_rate=0.04):
        K = len(full_itinary)
        out = np.array([full_itinary[k]["sidechains"][sidechain_name]["trigger"] for k in range(K)])

        # compute the attack delay kernel flipped (suited for convolution)
        attack_size = int(max(1, round(attack_s*fps)))
        attack_phase = np.linspace(0, 1, attack_size)
        rr = release_rate**(1/fps)
        release_size = np.log(0.01)/np.log(rr)
        release_phase = rr**(np.arange(release_size))
        kernel = np.concatenate((attack_phase, release_phase), axis=0)
        out = np.convolve(out, kernel, mode="same")
        out[out > 1] = 1
        return out

if __name__ == '__main__':
    itinary = Itinary(AnotherPlanetMap(), 10)