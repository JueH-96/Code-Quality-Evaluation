def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    d = {}
    for idx, num in enumerate(arr):
        if num not in d:
            d[num] = []
        d[num].append(idx)
    
    min_len = float('inf')
    for key, indices in d.items():
        count = len(indices)
        if count >= 2:
            for i in range(1, count):
                length = indices[i] - indices[i-1] + 1
                if length < min_len:
                    min_len = length
                    
    print(min_len if min_len != float('inf') else -1)

if __name__ == "__main__":
    main()