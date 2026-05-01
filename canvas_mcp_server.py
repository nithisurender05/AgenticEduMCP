import json
from mcp.server.fastmcp import FastMCP

# Initialize the Server
mcp = FastMCP("Canvas-Server")

# Helper function to load data
def load_data():
    with open("mock_canvas_data.json", "r") as f:
        return json.load(f)

@mcp.tool()
def get_student_grades(id: str, course_id: str) -> str:
    """Fetches the grade history for a specific student in a course."""
    data = load_data()
    for student in data["student_profiles"]:
        if student["id"] == id:
            for course in student["courses"]:
                if course["course_id"] == course_id:
                    return json.dumps(course["grades"])
    return "Student or Course not found."

@mcp.tool()
def get_professor_feedback(id: str, course_id: str) -> str:
    """Retrieves qualitative comments from instructors on past assignments."""
    data = load_data()
    for student in data["student_profiles"]:
        if student["id"] == id:
            for course in student["courses"]:
                if course["course_id"] == course_id:
                    return json.dumps(course["feedback"])
    return "Feedback not found."

@mcp.tool()
def fetch_syllabus_and_readings(course_id: str) -> str:
    """Returns the course schedule and recommended readings for a course."""
    data = load_data()
    for student in data["student_profiles"]:
        for course in student["courses"]:
            if course["course_id"] == course_id:
                return json.dumps(course["syllabus"])
    return "Syllabus not found."

if __name__ == "__main__":
    mcp.run()