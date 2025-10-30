def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    even_s = []
    even_d = []
    odd_s = []
    odd_d = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        s = x + y
        d = x - y
        if s % 2 == 0:
            even_s.append(s)
            even_d.append(d)
        else:
            odd_s.append(s)
            odd_d.append(d)
    
    def compute_sum(arr):
        n = len(arr)
        if n < 2:
            return 0
        arr.sort()
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + arr[i]
        total = 0
        for j in range(n):
            total += arr[j] * j - prefix[j]
        return total
    
    res = 0
    # Process even group
    s_even = compute_sum(even_s)
    d_even = compute_sum(even_d)
    res += (s_even + d_even) // 2
    # Process odd group
    s_odd = compute_sum(odd_s)
    d_odd = compute_sum(odd_d)
    res += (s_odd + d_odd) // 2
    print(res)

if __name__ == "__main__":
    main()