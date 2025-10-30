def solve():
    n = int(input())
    a = list(map(int, input().split()))
    if n <= 0:
        print(0)
        return
    count = 0
    for l_index in range(n):
        for r_index in range(l_index, n):
            subsequence = a[l_index:r_index+1]
            length = len(subsequence)
            if length <= 2:
                count += 1
            else:
                is_ap = True
                diff = subsequence[1] - subsequence[0]
                for i in range(2, length):
                    if subsequence[i] - subsequence[i-1] != diff:
                        is_ap = False
                        break
                if is_ap:
                    count += 1
    print(count)

if __name__ == '__main__':
    solve()