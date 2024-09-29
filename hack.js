

function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.bottom >= 0
    );
}

// Function to handle adding/removing the 'visible' class for all .fade-in elements
function handleScroll() {
    const fadeElements = document.querySelectorAll('.fade-in'); // Select all elements with fade-in class
    
    fadeElements.forEach(element => {
        if (isInViewport(element)) {
            element.classList.add('visible'); // Add visible class when in view
        } else {
            element.classList.remove('visible'); // Remove visible class when out of view
        }
    });
}

// Run the handleScroll function on page load and whenever the user scrolls
window.addEventListener('scroll', handleScroll);
window.addEventListener('load', handleScroll);


document.addEventListener("DOMContentLoaded", function(){
    Promise.all([
        d3.csv('/dataset/final_data_no_duplicates.csv')
    ]).then(function(value){
        wells_data = value

        console.log(wells_data);
        
    })
})
