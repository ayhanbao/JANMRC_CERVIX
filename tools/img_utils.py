import os


def norm_path(path):
    path = os.path.expanduser(path)
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.abspath(path)
    return path


def split_path(path):
    path = norm_path(path)
    root, name_ext = os.path.split(path)
    name, ext = os.path.splitext(name_ext)
    return root, name, ext


def image_list(path, exts=['.png', '.jpg']):
    path = norm_path(path)

    l = list()

    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue

            l.append(os.path.join(root, file))

    return l
