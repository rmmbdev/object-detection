import os

ROOT = '/code/'


def main():
    data_dir = os.path.join(ROOT, 'data')
    print(os.listdir(data_dir))


if __name__ == '__main__':
    main()
