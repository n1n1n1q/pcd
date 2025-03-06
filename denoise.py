import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point cloud denoising script')
    parser.add_argument('input', type=str, help='Input point cloud file')
    parser.add_argument('output', type=str, help='Output point cloud file')
    args = parser.parse_args()