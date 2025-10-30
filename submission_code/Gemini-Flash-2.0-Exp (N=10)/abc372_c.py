def solve():
    n, q = map(int, input().split())
    s = list(input())
    
    def count_abc(arr):
        count = 0
        for i in range(len(arr) - 2):
            if arr[i] == 'A' and arr[i+1] == 'B' and arr[i+2] == 'C':
                count += 1
        return count

    for _ in range(q):
        x, c = input().split()
        x = int(x)
        s[x-1] = c
        print(count_abc(s))

solve()