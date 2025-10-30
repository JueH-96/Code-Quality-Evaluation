def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    for s, t in zip(S, T):
        if s == t:
            continue
        elif (s == '1' and t == 'l') or (s == 'l' and t == '1'):
            continue
        elif (s == '0' and t == 'o') or (s == 'o' and t == '0'):
            continue
        else:
            print("No")
            return
    print("Yes")
    
if __name__ == '__main__':
    main()