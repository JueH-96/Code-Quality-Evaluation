def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    births = data[2:]
    
    # This dictionary will keep track of whether a family has already had a boy
    family_has_boy = [False] * (N + 1)
    
    index = 0
    for i in range(M):
        family = int(births[index]); index += 1
        sex = births[index]; index += 1
        
        if sex == 'M' and (not family_has_boy[family]):
            print("Yes")
            family_has_boy[family] = True
        else:
            print("No")

# Do not forget to call main()!
main()