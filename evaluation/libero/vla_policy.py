import cv2
import numpy as np
import requests


class LLaVAClient:
    def __init__(self, base_url='http://localhost:6800', request_timeout=600):
        self.base_url = base_url
        self.request_timeout = request_timeout

    def reset(self):
        # return requests.post(self.base_url+"/reset").json().get('response')
        pass

    def _request_frame(self, text, episode_first_frame, return_normalized=False, **kwargs):
        state = kwargs.pop('states', None)
        if state is not None:
            state = np.array(state)
            state = state.astype(np.float32)

        encoded_imgs = {}
        for name, img in kwargs.items():
            if img is not None:
                ret, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ret, "Image encode failed"
                encoded_img = encoded_img.tobytes()
            else:
                encoded_img = None
            encoded_imgs.update({name: encoded_img})

        ret = requests.post(
            self.base_url+"/process_frame",
            data={
                "text": text,
                'episode_first_frame': episode_first_frame,
                'return_normalized': str(return_normalized),
            },

            files={"image": encoded_imgs.pop('base_cam', None),
                    **({"states": ("states.npy", state.tobytes()) } if state is not None else {}),
                    },
            timeout=self.request_timeout,
        )
        ret.raise_for_status()
        return ret.json()

    def process_frame(self, text, episode_first_frame, **kwargs):
        return self._request_frame(text, episode_first_frame, **kwargs).get('response')

    def process_frame_with_diagnostics(self, text, episode_first_frame, **kwargs):
        return self._request_frame(
            text,
            episode_first_frame,
            return_normalized=True,
            **kwargs,
        )

    def experiment_control(self, action, **kwargs):
        ret = requests.post(
            self.base_url + "/experiment/control",
            json={"action": action, **kwargs},
            timeout=self.request_timeout,
        )
        ret.raise_for_status()
        return ret.json().get('response')

    def save_experiment_state(self, name):
        return self.experiment_control('save', name=name)

    def restore_experiment_state(self, name):
        return self.experiment_control('restore', name=name)

    def delete_experiment_state(self, name):
        return self.experiment_control('delete', name=name)

    def set_memory_retrieval(self, enabled):
        return self.experiment_control('set_retrieval', enabled=enabled)

    def set_experiment_seed(self, seed):
        return self.experiment_control('seed', seed=seed)

    def get_experiment_status(self):
        return self.experiment_control('status')
