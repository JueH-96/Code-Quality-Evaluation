def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()
    
    x = ['#'] * n
    
    def check(arr):
        for i in range(n):
            if arr[i] != s[i]:
                return False
        return True

    def backtrack(index, current_x):
        if index == n:
            if check(current_x):
                print("Yes")
                return True
            else:
                return False

        if current_x[index] == s[index]:
            if backtrack(index + 1, current_x):
                return True
        elif current_x[index] == '#':
            if index + m <= n:
                new_x = current_x[:]
                for i in range(m):
                    new_x[index + i] = t[i]
                if backtrack(index, new_x):
                    return True
        
        return False

    if backtrack(0, x):
        pass
    else:
        print("No")

solve()