import cv2
import numpy as np

# Chemin vers l'image source
image_path = 'p_1_2_9_129-thickbox_default-Maxi-Coca-15l.jpg'

# Lire l'image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Seuillage pour isoler l'objet
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

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

# Afficher le résultat
cv2.imshow('Objet Isolé sur 2 Bits', object_2bits)
cv2.waitKey(0)
cv2.destroyAllWindows()
