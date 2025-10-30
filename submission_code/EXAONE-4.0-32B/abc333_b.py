mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

def main():
    s = input().strip()
    t = input().strip()
    
    a, b = mapping[s[0]], mapping[s[1]]
    c, d_val = mapping[t[0]], mapping[t[1]]
    
    def get_min_step(x, y):
        diff = abs(x - y)
        return min(diff, 5 - diff)
    
    step1 = get_min_step(a, b)
    step2 = get_min_step(c, d_val)
    
    print("Yes" if step1 == step2 else "No")

if __name__ == "__main__":
    main()