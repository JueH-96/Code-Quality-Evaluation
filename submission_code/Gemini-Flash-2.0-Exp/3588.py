class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        alice_score = 0
        for char in s:
            if char == 'F':
                alice_score += 0
            elif char == 'W':
                alice_score += 0
            else:
                alice_score += 0

        def calculate_alice_points(alice_moves, bob_moves):
            points = 0
            for i in range(len(alice_moves)):
                if alice_moves[i] == 'F' and bob_moves[i] == 'E':
                    points += 1
                elif alice_moves[i] == 'W' and bob_moves[i] == 'F':
                    points += 1
                elif alice_moves[i] == 'E' and bob_moves[i] == 'W':
                    points += 1
            return points
        
        def calculate_bob_points(alice_moves, bob_moves):
            points = 0
            for i in range(len(alice_moves)):
                if bob_moves[i] == 'F' and alice_moves[i] == 'E':
                    points += 1
                elif bob_moves[i] == 'W' and alice_moves[i] == 'F':
                    points += 1
                elif bob_moves[i] == 'E' and alice_moves[i] == 'W':
                    points += 1
            return points

        count = 0
        
        def is_valid(moves):
            for i in range(len(moves) - 1):
                if moves[i] == moves[i+1]:
                    return False
            return True

        def generate_sequences(index, current_sequence):
            nonlocal count
            if index == n:
                if is_valid(current_sequence):
                    bob_score = calculate_bob_points(s, current_sequence)
                    alice_score = calculate_alice_points(s, current_sequence)
                    if bob_score > alice_score:
                        count = (count + 1) % (10**9 + 7)
                return
            
            generate_sequences(index + 1, current_sequence + 'F')
            generate_sequences(index + 1, current_sequence + 'W')
            generate_sequences(index + 1, current_sequence + 'E')

        generate_sequences(0, "")
        return count