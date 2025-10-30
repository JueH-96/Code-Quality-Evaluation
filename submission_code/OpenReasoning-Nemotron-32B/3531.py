class Solution:
	def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
		total_seconds_so_far = 0
		total_damage = 0
		enemies = sorted(zip(damage, health), key=lambda x: (-x[0], x[1]))
		for d, h in enemies:
			t = (h + power - 1) // power
			total_damage += d * (total_seconds_so_far + t)
			total_seconds_so_far += t
		return total_damage