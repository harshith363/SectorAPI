async function fetchSectorRecommendations() {
    try {
        const response = await fetch("http://127.0.0.1:5000/recommend-sectors");
        const data = await response.json();

        let recommendationHTML = "<ul>";
        data.forEach(sector => {
            recommendationHTML += `<li><b>${sector.sector}</b>: ${sector.probability}% confidence</li>`;
        });
        recommendationHTML += "</ul>";

        document.getElementById("sector-recommendations").innerHTML = recommendationHTML;
    } catch (error) {
        document.getElementById("sector-recommendations").innerHTML = "Error fetching recommendations.";
        console.error("Error fetching data:", error);
    }
}

fetchSectorRecommendations();
