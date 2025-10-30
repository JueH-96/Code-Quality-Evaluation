def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def is_pyramid(arr, k):
        if len(arr) != 2 * k - 1:
            return False
        for i in range(k):
            if arr[i] < i + 1:
                return False
        for i in range(k - 1):
            if arr[k + i] < k - 1 - i + 1:
                return False
        return True

    def get_pyramid_size(arr):
        for k in range(len(arr) // 2 + 1, 0, -1):
            if is_pyramid(arr, k):
                return k
        return 0

    ans = 0
    for i in range(1 << n):
        subsequence = []
        for j in range(n):
            if (i >> j) & 1:
                subsequence.append(a[j])
        
        if not subsequence:
            continue

        for start in range(len(subsequence) + 1):
            for end in range(start, len(subsequence) + 1):
                current = subsequence[start:end]
                if not current:
                    continue
                
                
                max_k = min(len(current) // 2 + 1, min(current) if current else 0)
                
                for k in range(max_k, 0, -1):
                    
                    temp = current[:]
                    
                    possible = True
                    
                    
                    for idx in range(k):
                        if temp[idx] < idx + 1:
                            possible = False
                            break
                    
                    for idx in range(k-1):
                        if temp[k+idx] < k - idx:
                            possible = False
                            break
                    
                    if len(temp) < 2*k -1:
                        possible = False
                    
                    if possible and len(temp) == 2*k -1:
                        ans = max(ans, k)
                        break
    print(ans)

solve()