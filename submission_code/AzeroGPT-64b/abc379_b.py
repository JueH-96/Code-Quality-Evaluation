from itertools import accumulate

def main() -> None:
    N, K = map(int, input().split())
    S = input()
    
    A = [0]
    B = ['X'] # remember: if there's no possibility to eat second role, still consume one
    
    K += 1
    for i, c in enumerate(S, 1):
        if c == 'O': 
            A.append(i)
        else: 
            B.append(-K)
            B.append(i)
    
    B += [N * 100]

    print(max((0,) + tuple(b - a for a, b in zip(A, B))))


if __name__ == '__main__':
    main()