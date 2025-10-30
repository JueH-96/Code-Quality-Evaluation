import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n < 2:
        print(0)
        return
        
    segments = []
    
    current = []
    for i in range(0, n-1, 2):
        if A[i] == A[i+1]:
            current.append(A[i])
        else:
            if current:
                segments.append(current)
                current = []
    if current:
        segments.append(current)
        
    current = []
    for i in range(1, n-1, 2):
        if A[i] == A[i+1]:
            current.append(A[i])
        else:
            if current:
                segments.append(current)
                current = []
    if current:
        segments.append(current)
        
    ans = 0
    for seg in segments:
        if not seg:
            continue
        left = 0
        freq = {}
        max_len = 0
        for right in range(len(seg)):
            num = seg[right]
            freq[num] = freq.get(num, 0) + 1
            while freq[num] > 1:
                left_num = seg[left]
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    del freq[left_num]
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        candidate = 2 * max_len
        if candidate > ans:
            ans = candidate
            
    print(ans)

if __name__ == "__main__":
    main()