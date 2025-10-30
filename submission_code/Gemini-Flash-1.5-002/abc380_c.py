def solve():
    n, k = map(int, input().split())
    s = input()
    
    blocks = []
    start = -1
    for i in range(n):
        if s[i] == '1':
            if start == -1:
                start = i
        else:
            if start != -1:
                blocks.append((start, i - 1))
                start = -1
    if start != -1:
        blocks.append((start, n - 1))

    
    if len(blocks) < k:
        print("Error: Not enough 1-blocks")
        return

    kth_block_start, kth_block_end = blocks[k-1]
    prev_block_end = blocks[k-2][1]

    
    result = list(s)
    
    block_len = kth_block_end - kth_block_start + 1
    
    for i in range(kth_block_start, kth_block_end + 1):
        result[i] = '0'

    for i in range(prev_block_end + 1, prev_block_end + 1 + block_len):
        result[i] = '1'

    print("".join(result))

solve()