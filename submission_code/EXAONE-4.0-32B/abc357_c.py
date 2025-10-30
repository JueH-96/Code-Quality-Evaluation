def main():
    N = int(input().strip())
    carpet = ['#']  # Level-0 carpet
    if N == 0:
        print('#')
        return
    for level in range(1, N+1):
        small_size = 3 ** (level - 1)
        total_size = 3 * small_size
        new_carpet = []
        for i in range(total_size):
            block_row = i // small_size
            idx = i % small_size
            if block_row == 1:
                mid_block = '.' * small_size
            else:
                mid_block = carpet[idx]
            new_row = carpet[idx] + mid_block + carpet[idx]
            new_carpet.append(new_row)
        carpet = new_carpet
    for row in carpet:
        print(row)

if __name__ == '__main__':
    main()