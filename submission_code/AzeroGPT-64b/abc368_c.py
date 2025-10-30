from typing import List

class GameMaster:
    def eliminateEnemies(self, enemies_health: List[int]) -> int:
        """
        Eliminates enemies in the game by attacking them with a variable T.
        When T is a multiple of 3, the attack deals 3 damage; otherwise, it deals 1 damage.
        Returns the value of T after all enemies' health becomes 0 or less.
        """
        current_t = 0
        for health in enemies_health:
            if health % 3 == 0:
                current_t += health * 2 // 3  # When health is a multiple of 3, effective damage is 2 per 3 T increments.
            else:
                # When health isn't a multiple of 3, at least one +1 damage must be given outside the group of three.
                current_t += (health // 3) * 3 + ((health % 3) > 0)
        return current_t

# The following is for testing the solution with given input data points.
# You can remove this section if you are using this code in a real-world scenario.
if __name__ == "__main__":
    game = GameMaster()
    import sys
    input_data = sys.stdin.readlines()
    N = int(input_data[0].strip())
    enemies_health = list(map(int, input_data[1].strip().split()))
    result = game.eliminateEnemies(enemies_health)
    print(result)