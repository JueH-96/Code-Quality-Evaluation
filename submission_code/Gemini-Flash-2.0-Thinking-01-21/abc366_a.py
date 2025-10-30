def solve():
    n, t, a = map(int, input().split())
    remaining_votes = n - (t + a)
    winning_threshold = (n + 1) // 2
    
    if (t + remaining_votes < winning_threshold) or (a + remaining_votes < winning_threshold):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()