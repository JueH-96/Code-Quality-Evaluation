# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    x = ["#"] * n

    def check():
        for i in range(n):
            if x[i] != "#" and x[i] != s[i]:
                return False
        return True

    def backtrack(idx):
        if idx == n:
            return check()

        if x[idx] != "#":
            return backtrack(idx + 1)
        
        
        if idx + m <= n:
            temp = x[idx:idx+m]
            x[idx:idx+m] = list(t)
            if backtrack(idx + 1):
                return True
            x[idx:idx+m] = temp
        
        
        return backtrack(idx+1)
        

    if backtrack(0):
        print("Yes")
    else:
        print("No")

solve()