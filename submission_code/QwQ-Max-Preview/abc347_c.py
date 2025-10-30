def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = int(input[1])
    B = int(input[2])
    D = list(map(int, input[3:3+N]))
    
    T = A + B
    if T == 0:
        print("No")
        return
    
    s_list = [(-d) % T for d in D]
    e_list = [(s + A) % T for s in s_list]
    
    L = max(s_list)
    R = min(e_list)
    
    if L < R:
        print("Yes")
        return
    else:
        possible = True
        for s, e in zip(s_list, e_list):
            if s < e:  # non-wrapping
                if s > 0:
                    possible = False
                    break
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()