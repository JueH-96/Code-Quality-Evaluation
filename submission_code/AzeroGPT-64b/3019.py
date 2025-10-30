class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        balance, wild = 0, 0
        for ch in moves:
            balance += 1 if ch == 'R' else -1 if ch == 'L' else 0
            wild += ch == '_'
        return abs(balance + wild)  # max if all wildcards are used to extend the current balance direction