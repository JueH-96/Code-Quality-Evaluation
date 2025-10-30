def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    results = []
    # Function to generate one-swap and no-swap possibilities for a string s.
    def can_be_made_abc(s):
        # if already sorted:
        if s == "abc":
            return True
        # Try all pairs of positions for swapping (only one swap allowed).
        s_list = list(s)
        for i in range(3):
            for j in range(i + 1, 3):
                t_list = s_list.copy()
                t_list[i], t_list[j] = t_list[j], t_list[i]
                if "".join(t_list) == "abc":
                    return True
        return False

    index = 1
    for _ in range(t):
        s = data[index]
        index += 1
        if can_be_made_abc(s):
            results.append("YES")
        else:
            results.append("NO")
    
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()