def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = int(input[1])
    t_list = list(map(int, input[2:2+n]))
    
    current_end = 0
    for t in t_list:
        start = max(t, current_end)
        finish = start + a
        print(finish)
        current_end = finish

if __name__ == "__main__":
    main()