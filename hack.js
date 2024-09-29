

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
        d3.csv('/dataset/final_data_no_duplicates.csv'),
        d3.json('/dataset/cb_2014_us_state_5m.geojson'),
        d3.json('/dataset/map2.geojson')
    ]).then(function(values){
        wells_data = values[0];
        navajo_region = values[2];
        navajo_state = values[1]

        wells_data.map(function(d){
            d['Longitude']= +d['Longitude'];
            d['Latitude']= + d['Latitude']
            d['well_id'] = +d['well_id']
            d['result'] = +d['result']
        })

        console.log(wells_data); 
        console.log(navajo_region);
        

        mapPlot()
    })
})

const size = d3.scaleLinear()
      .domain([0,3500])  // What's in the data
      .range([ 1, 15])  // Size in pixel

function mapPlot(){
    
    const width = 960;
    const height = 600;
    const colorScale = d3.scaleThreshold()
                        .domain([2,2.5,3,3.5,4,4.5,5,5.5])
                        .range(d3.schemeBlues[7]);
    
    const projection = d3.geoMercator()
                        .scale(1800)
                        .center([-470,35])
                        .translate([width / 2+50, height / 2]);

// Create SVG element
const svg = d3.select("svg")
let topo = navajo_state

svg.append("g")
        .selectAll("path")
        .data(topo.features)
        .enter()
        .append("path")
        .attr("d", d3.geoPath()
            .projection(projection)
        )
        .attr("fill","#E2DDDD9F")
        .style("stroke", "black")
          .attr("class", function(d){ return "States" } )
          .style("opacity", 1)

console.log(wells_data);


svg.selectAll("myCircles")
    .data(wells_data)
    .enter()
    .append("circle")
      .attr("cx", function(d){ return projection([d.long, d.lat])[0] })
      .attr("cy", function(d){ return projection([d.long, d.lat])[1] })
      .attr("r", d=> size(d.result))
      .style("fill","#39CFEDFF")
      .attr("stroke", '#000000FF')
      .attr("stroke-width", .2)
      .attr("fill-opacity", .5)
      


svg.append('text')
.attr('y',500)
        .attr('class', 'svg_text')
        .attr('x',width/2-200)
        .style('text-anchor','middle')
        .text('Water wells in Navajo region' )
 
// state names
svg.append('text')
.attr('y',345)
        .attr('class', 'svg_map_text')
        .attr('x',width/2)
        .style('text-anchor','middle')
        .text('Arizona' )

svg.append('text')
.attr('y',145)
        .attr('class', 'svg_map_text')
        .attr('x',width/2)
        .style('text-anchor','middle')
        .text('Utah' )

svg.append('text')
.attr('y',145)
        .attr('class', 'svg_map_text')
        .attr('x',width/2+200)
        .style('text-anchor','middle')
        .text('Colarado' )

svg.append('text')
.attr('y',345)
        .attr('class', 'svg_map_text')
        .attr('x',width/2+200)
        .style('text-anchor','middle')
        .text('New Mexico' )


    
}
