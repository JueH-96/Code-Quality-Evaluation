# YOUR CODE HERE
n, m = map(int, input().split())
s = input()
t = input()

x = ['#'] * n

def solve():
    q = [list(x)]
    visited = {tuple(x)}

    while q:
        curr_x = q.pop(0)
        if "".join(curr_x) == s:
            return True

        for i in range(n - m + 1):
            next_x = list(curr_x)
            next_x[i:i+m] = list(t)
            next_x_tuple = tuple(next_x)
            if next_x_tuple not in visited:
                visited.add(next_x_tuple)
                q.append(next_x)
    return False

if solve():
    print("Yes")
else:
    print("No")