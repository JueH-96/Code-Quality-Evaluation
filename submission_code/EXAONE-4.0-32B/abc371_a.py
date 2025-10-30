def main():
    data = input().split()
    s_ab = data[0]
    s_ac = data[1]
    s_bc = data[2]
    
    olders_A = (1 if s_ab == '<' else 0) + (1 if s_ac == '<' else 0)
    olders_B = (1 if s_ab == '>' else 0) + (1 if s_bc == '<' else 0)
    olders_C = (1 if s_ac == '>' else 0) + (1 if s_bc == '>' else 0)
    
    if olders_A == 1:
        print('A')
    elif olders_B == 1:
        print('B')
    else:
        print('C')

if __name__ == '__main__':
    main()