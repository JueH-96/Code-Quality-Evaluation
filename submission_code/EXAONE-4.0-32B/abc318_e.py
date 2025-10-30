def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    left_freq = [0] * (n + 1)
    right_freq = [0] * (n + 1)
    
    for num in A:
        right_freq[num] += 1
    
    T = 0
    ans = 0
    
    for j in range(n):
        x = A[j]
        T -= left_freq[x]
        right_freq[x] -= 1
        
        total_excluding_x = T - (left_freq[x] * right_freq[x])
        ans += total_excluding_x
        
        T += right_freq[x]
        left_freq[x] += 1
        
    print(ans)

if __name__ == "__main__":
    main()