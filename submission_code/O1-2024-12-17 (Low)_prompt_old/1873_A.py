def solve():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    pointer = 1

    for _ in range(t):
        s = data[pointer]
        pointer += 1

        # If it's already 'abc', print YES
        if s == "abc":
            print("YES")
            continue
        
        # Check if by one swap we can get 'abc'
        # There are exactly three positions: (0,1), (0,2), and (1,2) to try swapping.
        possible = False
        s_list = list(s)
        for i in range(3):
            for j in range(i+1, 3):
                swapped = s_list[:]
                swapped[i], swapped[j] = swapped[j], swapped[i]
                if "".join(swapped) == "abc":
                    possible = True
                    break
            if possible:
                break
        
        if possible:
            print("YES")
        else:
            print("NO")

def main():
    solve()

if __name__ == "__main__":
    main()