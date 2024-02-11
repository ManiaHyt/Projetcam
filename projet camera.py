import cv2
import numpy as np

# Chemin vers l'image source
image_path = 'yuno_levi_jardin.jpg'

# Lire l'image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Seuillage pour isoler l'objet
_, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

# Trouver les contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Supposons que l'objet principal est le plus grand contour
contour = max(contours, key=cv2.contourArea)

# Créer un masque pour l'objet
mask = np.zeros_like(gray)
cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

# Appliquer le masque pour isoler l'objet
object_only = cv2.bitwise_and(gray, gray, mask=mask)

# Réduire l'objet isolé à 2 bits
object_2bits = (object_only // 64) * 85

# Redimensionner l'image pour l'affichage
# Définissez les nouvelles dimensions (largeur, hauteur)
new_width = 800
new_height = int((new_width / object_2bits.shape[1]) * object_2bits.shape[0])
resized_image = cv2.resize(object_2bits, (new_width, new_height))

# Afficher le résultat redimensionné
cv2.imshow('Objet Isole sur 2 Bits - Redimensionne', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
