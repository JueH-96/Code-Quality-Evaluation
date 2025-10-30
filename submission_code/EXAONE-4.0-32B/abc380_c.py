import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, k = map(int, data[0].split())
    s = data[1].strip()
    
    blocks = []
    i = 0
    n_val = len(s)
    while i < n_val:
        if s[i] == '1':
            start = i
            j = i
            while j < n_val and s[j] == '1':
                j += 1
            end = j - 1
            blocks.append((start, end))
            i = j
        else:
            i += 1
            
    prev_block = blocks[k-2]
    curr_block = blocks[k-1]
    l_prev, r_prev = prev_block
    l_curr, r_curr = curr_block
    L = r_curr - l_curr + 1
    
    part1 = s[:r_prev+1]
    part2 = '1' * L
    num_zeros = r_curr - r_prev - L
    part3 = '0' * num_zeros
    part4 = s[r_curr+1:]
    
    result = part1 + part2 + part3 + part4
    print(result)

if __name__ == "__main__":
    main()