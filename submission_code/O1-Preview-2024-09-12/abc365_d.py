# YOUR CODE HERE
import sys

def main():
    import sys
    import threading
    def solve():
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        BeatingMove = {'R':'P', 'P':'S', 'S':'R'}
        PrevMove = None
        Wins = 0
        for i in range(N):
            AokiMove = S[i]
            beat = BeatingMove[AokiMove]
            draw = AokiMove
            if beat != PrevMove:
                Move = beat
                Wins +=1
            else:
                Move = draw
            PrevMove = Move
        print(Wins)
    threading.Thread(target=solve).start()
main()