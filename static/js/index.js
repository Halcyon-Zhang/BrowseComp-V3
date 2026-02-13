window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Experiment Analysis: one panel at a time, custom slider (no bulma carousel)
var experimentAnalysisIndex = 0;
var experimentAnalysisTotal = 3;

function experimentAnalysisUpdate() {
	var inner = document.getElementById('experiment-panels-inner');
	var dots = document.querySelectorAll('#experiment-pagination .experiment-dot');
	if (!inner || !dots.length) return;
	// Each panel is 1/3 of inner width; move by (index * 100/3)%
	var percent = (experimentAnalysisIndex * 100 / experimentAnalysisTotal);
	inner.style.transform = 'translateX(-' + percent + '%)';
	dots.forEach(function(dot, i) {
		dot.classList.toggle('is-active', i === experimentAnalysisIndex);
	});
}

function experimentAnalysisPrev() {
	experimentAnalysisIndex = (experimentAnalysisIndex - 1 + experimentAnalysisTotal) % experimentAnalysisTotal;
	experimentAnalysisUpdate();
}

function experimentAnalysisNext() {
	experimentAnalysisIndex = (experimentAnalysisIndex + 1) % experimentAnalysisTotal;
	experimentAnalysisUpdate();
}

function experimentAnalysisGo(i) {
	experimentAnalysisIndex = Number(i);
	experimentAnalysisUpdate();
}

// Experiment Analysis: init on DOMContentLoaded (no jQuery dependency) + delegate clicks on document
function initExperimentAnalysis() {
	experimentAnalysisUpdate();
	// Delegate all experiment-nav and experiment-dot clicks from document (always works)
	document.addEventListener('click', function(e) {
		if (e.target.closest && e.target.closest('#experiment-nav-prev')) {
			e.preventDefault();
			experimentAnalysisPrev();
			return;
		}
		if (e.target.closest && e.target.closest('#experiment-nav-next')) {
			e.preventDefault();
			experimentAnalysisNext();
			return;
		}
		var dot = e.target.closest && e.target.closest('.experiment-dot');
		if (dot && dot.closest('#experiment-pagination') && dot.dataset.index !== undefined) {
			e.preventDefault();
			experimentAnalysisGo(parseInt(dot.dataset.index, 10));
		}
	});
}
if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', initExperimentAnalysis);
} else {
	initExperimentAnalysis();
}

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize carousels (general)
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Experiment Analysis already inited via DOMContentLoaded above
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

})
