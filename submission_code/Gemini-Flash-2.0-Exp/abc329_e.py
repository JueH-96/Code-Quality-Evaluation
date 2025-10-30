def solve():
    n, m = map(int, input().split())
    s = input()
    t = input()

    def check(indices):
        x = ['#'] * n
        for index in indices:
            for i in range(m):
                x[index + i] = t[i]
        return "".join(x) == s

    def find_all_indices(target, string):
        indices = []
        for i in range(len(string) - len(target) + 1):
            if string[i:i+len(target)] == target:
                indices.append(i)
        return indices

    possible_indices = find_all_indices(t, s)
    
    import itertools
    
    for r in range(n + 1):
        for indices in itertools.combinations(range(n - m + 1), r):
            valid = True
            for index in indices:
                
                temp_x = ['#'] * n
                for ind in indices:
                    for i in range(m):
                        temp_x[ind + i] = t[i]
                
                if "".join(temp_x) == s:
                    print("Yes")
                    return
    print("No")

solve()