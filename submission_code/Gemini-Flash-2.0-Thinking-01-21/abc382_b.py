def solve():
    n, d = map(int, input().split())
    s = input()
    box_states = list(s)
    for _ in range(d):
        rightmost_cookie_index = -1
        for i in range(n - 1, -1, -1):
            if box_states[i] == '@':
                rightmost_cookie_index = i
                break
        if rightmost_cookie_index != -1:
            box_states[rightmost_cookie_index] = '.'
    print("".join(box_states))

if __name__ == '__main__':
    solve()