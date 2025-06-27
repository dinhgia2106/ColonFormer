try:
    import mmcv
    MMCV_AVAILABLE = True
except ImportError:
    mmcv = None
    MMCV_AVAILABLE = False
    # print("Warning: mmcv not found, running in standalone mode")

from .version import __version__, version_info

# Only check mmcv version if available
if MMCV_AVAILABLE:
    MMCV_MIN = '1.1.4'
    MMCV_MAX = '1.3.0'

    def digit_version(version_str):
        digit_version = []
        for x in version_str.split('.'):
            if x.isdigit():
                digit_version.append(int(x))
            elif x.find('rc') != -1:
                patch_version = x.split('rc')
                digit_version.append(int(patch_version[0]) - 1)
                digit_version.append(int(patch_version[1]))
        return digit_version

    mmcv_min_version = digit_version(MMCV_MIN)
    mmcv_max_version = digit_version(MMCV_MAX)
    mmcv_version = digit_version(mmcv.__version__)

    # Only warn about incompatible version, don't crash
    if not (mmcv_min_version <= mmcv_version <= mmcv_max_version):
        print(f'Warning: MMCV=={mmcv.__version__} is incompatible. '
              f'Recommended: mmcv>={MMCV_MIN}, <={MMCV_MAX}')

__all__ = ['__version__', 'version_info']
