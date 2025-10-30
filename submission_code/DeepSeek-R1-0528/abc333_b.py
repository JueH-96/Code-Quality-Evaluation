def main():
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    first_line = input().strip()
    second_line = input().strip()
    
    s1, s2 = first_line[0], first_line[1]
    t1, t2 = second_line[0], second_line[1]
    
    a1 = mapping[s1]
    a2 = mapping[s2]
    diff1 = abs(a1 - a2)
    step1 = min(diff1, 5 - diff1)
    
    b1 = mapping[t1]
    b2 = mapping[t2]
    diff2 = abs(b1 - b2)
    step2 = min(diff2, 5 - diff2)
    
    if step1 == step2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()