import itertools

def main():
    S_AB, S_AC, S_BC = input().split()
    brothers = ['A', 'B', 'C']
    for perm in itertools.permutations(brothers):
        if ( (S_AB == '<' and perm.index('A') < perm.index('B')) or
             (S_AB == '>' and perm.index('A') > perm.index('B')) ):
            if ( (S_AC == '<' and perm.index('A') < perm.index('C')) or
                 (S_AC == '>' and perm.index('A') > perm.index('C')) ):
                if ( (S_BC == '<' and perm.index('B') < perm.index('C')) or
                     (S_BC == '>' and perm.index('B') > perm.index('C')) ):
                    print(perm[1])
                    return

if __name__ == "__main__":
    main()