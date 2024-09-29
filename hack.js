

function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.bottom >= 0
    );
}

// Function to add/remove the 'visible' class based on the element's visibility
function handleScroll() {
    const element = document.querySelector('#future_act'); // Select the Sustainability section
    if (isInViewport(element)) {
        element.classList.add('visible'); // Add visible class when in view
    } else {
        element.classList.remove('visible'); // Remove visible class when out of view
    }
}

// Run the handleScroll function on page load and whenever the user scrolls
window.addEventListener('scroll', handleScroll);
window.addEventListener('load', handleScroll); // 