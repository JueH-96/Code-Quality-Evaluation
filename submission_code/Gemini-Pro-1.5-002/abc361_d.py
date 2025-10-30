# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    t = input()

    q = [(s, 0)]
    visited = {s}

    while q:
        curr_s, moves = q.pop(0)

        if curr_s == t:
            print(moves)
            return

        for i in range(n + 1):
            if curr_s[i:i+2].count('.') == 0:
                
                empty_start = -1
                for j in range(n + 2):
                    if curr_s[j:j+2] == '..':
                        empty_start = j
                        break
                
                if empty_start != -1:
                    next_s = list(curr_s)
                    next_s[empty_start] = curr_s[i]
                    next_s[empty_start + 1] = curr_s[i+1]
                    next_s[i] = '.'
                    next_s[i+1] = '.'
                    next_s = "".join(next_s)

                    if next_s not in visited:
                        visited.add(next_s)
                        q.append((next_s, moves + 1))

    print(-1)

solve()