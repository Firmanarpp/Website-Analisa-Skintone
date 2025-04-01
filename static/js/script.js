document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const hamburger = document.querySelector(".hamburger");
  const navLinks = document.querySelector(".nav-links");
  const navLinksItems = document.querySelectorAll(".nav-links a");
  const uploadArea = document.getElementById("upload-area");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.querySelector(".browse-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const resultsSection = document.getElementById("results-section");
  const confidenceEl = document.getElementById("confidence");
  const mstResult = document.getElementById("mst-result");
  const skinTypeGroup = document.getElementById("skin-type-group");
  const recommendedColors = document.getElementById("recommended-colors");
  const avoidColors = document.getElementById("avoid-colors");
  const loadingOverlay = document.getElementById("loading-overlay");
  const mstColors = document.querySelectorAll(".mst-color");
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const changeImageBtn = document.getElementById("change-image-btn");
  const accordionBtns = document.querySelectorAll(".accordion-button");

  let selectedFile = null;

  // Mobile menu toggle
  if (hamburger) {
    hamburger.addEventListener("click", () => {
      hamburger.classList.toggle("active");
      navLinks.classList.toggle("active");
    });
  }

  // Close mobile menu when clicking on links
  navLinksItems.forEach((item) => {
    item.addEventListener("click", () => {
      hamburger.classList.remove("active");
      navLinks.classList.remove("active");
    });
  });

  // Add event listeners for file upload
  if (uploadArea) {
    uploadArea.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", handleFileSelect);
    uploadArea.addEventListener("dragover", handleDragOver);
    uploadArea.addEventListener("dragleave", handleDragLeave);
    uploadArea.addEventListener("drop", handleDrop);
  }

  if (browseBtn) {
    browseBtn.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent the event from bubbling up to uploadArea
      fileInput.click();
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyzeImage);
  }

  if (changeImageBtn) {
    changeImageBtn.addEventListener("click", () => {
      fileInput.click();
    });
  }

  // Tab functionality
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      // Remove active class from all buttons and panes
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      // Add active class to clicked button and corresponding pane
      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      document.getElementById(`${tabId}-tab`).classList.add("active");
    });
  });

  // Accordion functionality
  accordionBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const accordionItem = btn.parentElement;
      accordionItem.classList.toggle("active");
      btn.classList.toggle("active");
    });
  });

  // Handle file selection from input
  function handleFileSelect(e) {
    selectedFile = e.target.files[0];
    if (selectedFile) {
      displayImagePreview(selectedFile);
      analyzeBtn.disabled = false;
    }
  }

  // Handle drag over
  function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add("active");
  }

  // Handle drag leave
  function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove("active");
  }

  // Handle file drop
  function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove("active");

    if (e.dataTransfer.files.length) {
      selectedFile = e.dataTransfer.files[0];
      fileInput.files = e.dataTransfer.files;
      displayImagePreview(selectedFile);
      analyzeBtn.disabled = false;
    }
  }

  // Display image preview
  function displayImagePreview(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
      imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
      previewContainer.style.display = "block";
    };

    reader.readAsDataURL(file);
  }

  // Analyze the image
  function analyzeImage() {
    if (!selectedFile) return;

    // Show loading overlay
    loadingOverlay.classList.add("active");

    const formData = new FormData();
    formData.append("file", selectedFile);

    fetch("/analyze", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        loadingOverlay.classList.remove("active");

        if (data.error) {
          showError(data.error);
          return;
        }

        displayResults(data);
      })
      .catch((error) => {
        loadingOverlay.classList.remove("active");
        showError("An error occurred during analysis. Please try again.");
        console.error("Error:", error);
      });
  }

  // Update the displayResults function in your script.js file:

  function displayResults(data) {
    // Show results section
    resultsSection.style.display = "block";

    // Set confidence
    confidenceEl.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;

    // Update MST scale visualization
    mstColors.forEach((el) => el.classList.remove("active"));
    const activeMst = document.querySelector(
      `.mst-color[data-mst="${data.prediction}"]`
    );
    if (activeMst) {
      activeMst.classList.add("active");
    }

    // Update result text
    mstResult.textContent = `Your skin tone is classified as ${data.prediction} on the Monk Skin Tone Scale`;

    // Remove any existing face detection info
    const existingFaceInfo = document.getElementById("face-detection-info");
    if (existingFaceInfo) {
      existingFaceInfo.remove();
    }

    // Create new face detection info
    const faceDetectionInfo = document.createElement("div");
    faceDetectionInfo.id = "face-detection-info";
    faceDetectionInfo.className = "face-detection-info";

    if (data.face_detected) {
      faceDetectionInfo.innerHTML = `
            <div class="detection-badge success">
                <i class="fas fa-check-circle"></i> Face detected
            </div>
            <p>Skin tone analyzed from facial region for higher accuracy</p>
            <div class="processed-image-container">
                <h4>Face Detection</h4>
                <div class="processed-image">
                    <img src="${
                      data.processed_image_url
                    }?t=${new Date().getTime()}" alt="Face Detection">
                </div>
            </div>
        `;
    } else {
      faceDetectionInfo.innerHTML = `
            <div class="detection-badge warning">
                <i class="fas fa-exclamation-triangle"></i> No face detected
            </div>
            <p>Entire image was analyzed. For better results, please upload a clear photo of your face.</p>
        `;
    }

    // Insert after mst-result element
    mstResult.parentNode.insertBefore(faceDetectionInfo, mstResult.nextSibling);

    // Set skin tone group
    if (skinTypeGroup) {
      skinTypeGroup.textContent = `Skin Tone Group: ${capitalizeFirstLetter(
        data.skin_tone_group
      )}`;
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth" });

    // Display clothing recommendations
    displayClothingRecommendations(data.recommendations);
  }

  // Display clothing recommendations
  function displayClothingRecommendations(recommendations) {
    // Clear previous recommendations
    recommendedColors.innerHTML = "";
    avoidColors.innerHTML = "";

    // Add recommended colors
    recommendations.recommended.forEach((color) => {
      const colorChip = createColorChip(color.name, color.hex);
      recommendedColors.appendChild(colorChip);
    });

    // Add colors to avoid
    recommendations.avoid.forEach((color) => {
      const colorChip = createColorChip(color.name, color.hex);
      avoidColors.appendChild(colorChip);
    });
  }

  // Create color chip element
  function createColorChip(name, hex) {
    const chip = document.createElement("div");
    chip.className = "color-chip";

    const colorPreview = document.createElement("div");
    colorPreview.className = "color-preview";
    colorPreview.style.backgroundColor = hex;

    const colorName = document.createElement("span");
    colorName.textContent = name;

    chip.appendChild(colorPreview);
    chip.appendChild(colorName);

    return chip;
  }

  // Show error
  function showError(message) {
    alert(message);
  }

  // Helper function to capitalize first letter
  function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();

      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);

      if (targetElement) {
        const headerOffset = 100;
        const elementPosition = targetElement.getBoundingClientRect().top;
        const offsetPosition =
          elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: "smooth",
        });
      }
    });
  });
});
