def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    results = []
    
    for i in range(1, t + 1):
        n = int(data[i])
        
        # Vanya starts and if n is already divisible by 3, he wins immediately.
        if n % 3 == 0:
            results.append("First")
        else:
            # If not, consider the next move by Vanya:
            # He can make n to n+1 or n-1.
            # If n+1 or n-1 is divisible by 3, he can win in his first move.
            if (n + 1) % 3 == 0 or (n - 1) % 3 == 0:
                results.append("First")
            else:
                # If neither n+1 nor n-1 is divisible by 3, then:
                # - Vanya makes n to n+1 or n-1 (neither are divisible by 3)
                # - Vova makes another move, and can only make it to n, n+2 or n-2
                # This cycle continues until Vanya's turn:
                # - If it's Vanya's turn and he can make the number divisible by 3, he wins.
                # - If 10 moves pass and he hasn't won, Vova wins.
                
                # Since the game starts with n and Vanya can't win immediately,
                # we simulate the next few moves:
                # - Vanya: n -> n+1 (or n-1)
                # - Vova: n+1 -> n+2 (or n)
                # - Vanya: n+2 -> n+3 (or n+1)
                # - Vova: n+3 -> n+4 (or n+2)
                # - Vanya: n+4 -> n+5 (or n+3)
                # - Vova: n+5 -> n+6 (or n+4)
                # - Vanya: n+6 -> n+7 (or n+5)
                # - Vova: n+7 -> n+8 (or n+6)
                # - Vanya: n+8 -> n+9 (or n+7)
                # - Vova: n+9 -> n+10 (or n+8)
                # - Vanya: n+10 -> n+11 (or n+9)
                
                # Vanya can win on his 1st, 3rd, 5th, 7th, 9th move if he can make the number divisible by 3.
                # We need to check if any of these moves can make the number divisible by 3.
                
                # If n % 3 == 1, then:
                # - n+2, n+5, n+8 are the numbers Vanya can make divisible by 3 on his turns.
                # If n % 3 == 2, then:
                # - n+1, n+4, n+7, n+10 are the numbers Vanya can make divisible by 3 on his turns.
                
                if n % 3 == 1:
                    # Vanya's moves: n+2, n+5, n+8
                    if (n + 2) % 3 == 0 or (n + 5) % 3 == 0 or (n + 8) % 3 == 0:
                        results.append("First")
                    else:
                        results.append("Second")
                elif n % 3 == 2:
                    # Vanya's moves: n+1, n+4, n+7, n+10
                    if (n + 1) % 3 == 0 or (n + 4) % 3 == 0 or (n + 7) % 3 == 0 or (n + 10) % 3 == 0:
                        results.append("First")
                    else:
                        results.append("Second")
    
    for result in results:
        print(result)