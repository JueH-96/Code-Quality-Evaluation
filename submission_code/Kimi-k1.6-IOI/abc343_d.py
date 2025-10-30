def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    T = int(data[idx])
    idx += 1
    
    scores = [0] * N
    freq = {0: N}
    result = []
    
    for _ in range(T):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        A -= 1  # convert to 0-based index
        
        old = scores[A]
        # Decrement old score frequency
        if freq[old] == 1:
            del freq[old]
        else:
            freq[old] -= 1
        
        new = old + B
        scores[A] = new
        # Increment new score frequency
        if new in freq:
            freq[new] += 1
        else:
            freq[new] = 1
        
        result.append(str(len(freq)))
    
    print('
'.join(result))

if __name__ == '__main__':
    main()