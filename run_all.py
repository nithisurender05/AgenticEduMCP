import json
import asyncio
from run_single_experiment import run_experiment

async def main():
    with open("mock_canvas_data.json") as f:
        data = json.load(f)

    results = []

    for student in data["student_profiles"]:
        try:
            print(f"\nRunning for {student['name']}...")
            res = await run_experiment(student)
        except Exception as e:
            print(f"[FAILED STUDENT {student['id']}] {e}")
            res = {"student_id": student["id"], "error": str(e)}

        results.append(res)

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll results saved to results.json")

if __name__ == "__main__":
    asyncio.run(main())