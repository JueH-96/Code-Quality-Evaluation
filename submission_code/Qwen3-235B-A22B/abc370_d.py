import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    Q = int(data[2])
    queries = []
    index = 3
    for _ in range(Q):
        r = int(data[index])
        c = int(data[index+1])
        queries.append((r, c))
        index += 2

    destroyed = set()
    total = H * W

    for r, c in queries:
        pos = (r, c)
        if pos in destroyed:
            # Up
            nr = r - 1
            found = False
            while nr >= 1:
                new_pos = (nr, c)
                if new_pos not in destroyed:
                    destroyed.add(new_pos)
                    total -= 1
                    found = True
                    break
                nr -= 1
            # Down
            nr = r + 1
            found = False
            while nr <= H:
                new_pos = (nr, c)
                if new_pos not in destroyed:
                    destroyed.add(new_pos)
                    total -= 1
                    found = True
                    break
                nr += 1
            # Left
            nc = c - 1
            found = False
            while nc >= 1:
                new_pos = (r, nc)
                if new_pos not in destroyed:
                    destroyed.add(new_pos)
                    total -= 1
                    found = True
                    break
                nc -= 1
            # Right
            nc = c + 1
            found = False
            while nc <= W:
                new_pos = (r, nc)
                if new_pos not in destroyed:
                    destroyed.add(new_pos)
                    total -= 1
                    found = True
                    break
                nc += 1
        else:
            destroyed.add(pos)
            total -= 1

    print(total)

if __name__ == "__main__":
    main()