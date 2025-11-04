import os
import psutil
import numpy as np
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

def split_into_blocks(image, block_size=(2, 2)):
    blocks = []
    image = image.squeeze(0).numpy()
    rows, cols = image.shape
    block_height, block_width = block_size
    for row in range(0, rows, block_height):
        for col in range(0, cols, block_width):
            block = image[row:row + block_height, col:col + block_width].reshape(-1, )
            blocks.append(block)

    return np.array(blocks).T

def restore_image(blocks, grid_size=(14, 14), block_size=(2, 2)):
    grid_rows, grid_cols = grid_size
    block_height, block_width = block_size
    restored_image = np.zeros((grid_rows * block_height, grid_cols * block_width))
    for i in range(grid_rows):
        for j in range(grid_cols):
            block_idx = i * grid_cols + j
            restored_image[
            i * block_height: (i + 1) * block_height,
            j * block_width: (j + 1) * block_width
            ] = blocks[:, block_idx].reshape(block_size)

    return restored_image

MAX_LENGTH = 2500
def encode_labels(label_list, code2idx):
    y = np.zeros(len(code2idx))
    for code in label_list:
        if code in code2idx:
            y[code2idx[code]] = 1
    return y

def encode_text(text, w2ind, max_len=MAX_LENGTH):
    tokens = text.split()
    seq = [w2ind.get(tok, 0) for tok in tokens]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [0] * (max_len - len(seq))
    return seq