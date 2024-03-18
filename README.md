## Moodfy Web Application
Moodfy is a web application designed to identify the prevalent emotion of a Spotify playlist, pinpoint tracks within the playlist that do not align with this emotion using outlier detection techniques and provides recommendations based on musical genre and predominant emotion. The application leverages the Spotify API for playlist analysis and visualization.

## Technologies Used
**Python:** The primary programming language for developing the Moodfy application.

**Requests:** Python library for making HTTP requests to the Spotify API.

**Pandas:** Data manipulation and analysis library used for processing data obtained from the Spotify API.

**NumPy:** Fundamental package for numerical computing in Python.

**scikit-learn (sklearn):** Machine learning library in Python, utilized for outlier detection techniques.

**Plotly:** Python graphing library for creating interactive visualizations.

**Amazon RDS:** Database managing.

**Dash:** A productive Python framework for building web applications.

**Render:** Platform used to host and display the web app.

**Spotify API:** Spotify's Web API used to fetch playlist data and track information.

## Access the Moodfy Web App
You can access the Moodfy web application using the following link: [Moodfy Web App](https://moodfy-v2.onrender.com/)

## How Moodfy Works
Moodfy analyzes a Spotify playlist to determine the prevalent emotion based on the tracks within it. It then employs outlier detection techniques to identify tracks that significantly differ from this prevalent emotion and provides recommendations based on musical genre and predominant emotion. The results are displayed through an interactive dashboard using Plotly and Dash, providing users with insights and recommendations into the emotional composition of their playlists.
